package org.dmieter.sch.vm.simulation;

import org.dmieter.sch.vm.data.TaskHistory;
import org.dmieter.sch.vm.data.TaskPeriod;
import org.dmieter.sch.vm.data.VirtualMachine;

import java.util.List;

public class Simulation {
    int time = 0;

    List<TaskPeriod> taskFlow;
    List<TaskHistory> history;
    List<VirtualMachine> vmTemplates;

    List<VirtualMachine> activeVMs;
    List<VirtualMachine> releasedVMs;

    public void setTime(int time) {
        if (this.time > time) {
            throw new IllegalStateException("Can't go back in time");
        }

        this.time = time;
    }

    public VirtualMachine createVirtualMachine(VirtualMachine template) {
        return new VirtualMachine(0);
    }

    public void destroy(VirtualMachine virtualMachine) {
        if(virtualMachine.status == VirtualMachine.Status.destroyed) {
            throw new IllegalStateException("Can't destroy already destroyed VM " + virtualMachine.id);
        }

        activeVMs.remove(virtualMachine);
        virtualMachine.setStatus(VirtualMachine.Status.destroyed);
        virtualMachine.setTimeDestroyed(time);
        releasedVMs.add(virtualMachine);
    }
}
